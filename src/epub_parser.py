import os
import warnings
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")


def extract_chapters_from_epub(epub_file):
    """Extract chapters from epub file using ebooklib's metadata and TOC."""
    if not os.path.exists(epub_file):
        raise FileNotFoundError(f"EPUB file not found: {epub_file}")

    book = epub.read_epub(epub_file)
    chapters = []

    def get_chapter_content(soup, start_id, next_id=None):
        """Extract content between two fragment IDs"""
        content = []
        start_elem = soup.find(id=start_id)

        if not start_elem:
            return ""

        # Skip the heading itself if it's a heading
        if start_elem.name in ["h1", "h2", "h3", "h4"]:
            current = start_elem.find_next_sibling()
        else:
            current = start_elem

        while current:
            # Stop if we hit the next chapter
            if next_id and current.get("id") == next_id:
                break
            # Stop if we hit another chapter heading
            if (
                current.name in ["h1", "h2", "h3"]
                and "chapter" in current.get_text().lower()
            ):
                break
            content.append(current.get_text())
            current = current.find_next_sibling()

        return "\n".join(content).strip()

    def process_toc_items(items, depth=0):
        processed = []
        for i, item in enumerate(items):
            if isinstance(item, tuple):
                section_title, section_items = item
                processed.extend(process_toc_items(section_items, depth + 1))
            elif isinstance(item, epub.Link):
                pass

                # Skip if title suggests it's front matter
                if item.title.lower() in [
                    "copy",
                    "copyright",
                    "title page",
                    "cover",
                ] or item.title.lower().startswith("by"):
                    continue

                # Extract the file name and fragment from href
                href_parts = item.href.split("#")
                file_name = href_parts[0]
                fragment_id = href_parts[1] if len(href_parts) > 1 else None

                # Find the document
                doc = next(
                    (
                        doc
                        for doc in book.get_items_of_type(ITEM_DOCUMENT)
                        if doc.file_name.endswith(file_name)
                    ),
                    None,
                )

                if doc:
                    content = doc.get_content().decode("utf-8")
                    soup = BeautifulSoup(content, "html.parser")

                    # If no fragment ID, get whole document content
                    if not fragment_id:
                        text_content = soup.get_text().strip()
                    else:
                        # Get the next fragment ID if available
                        next_item = items[i + 1] if i + 1 < len(items) else None
                        next_fragment = None
                        if isinstance(next_item, epub.Link):
                            next_href_parts = next_item.href.split("#")
                            if (
                                next_href_parts[0] == file_name
                                and len(next_href_parts) > 1
                            ):
                                next_fragment = next_href_parts[1]

                        # Extract content between fragments
                        text_content = get_chapter_content(
                            soup, fragment_id, next_fragment
                        )

                    if text_content:
                        chapters.append(
                            {
                                "title": item.title,
                                "content": text_content,
                                "order": len(processed) + 1,
                            }
                        )
                        processed.append(item)
        return processed

    # Process the table of contents
    process_toc_items(book.toc)

    # If no chapters were found through TOC, try processing all documents
    if not chapters:
        # Get all document items sorted by file name
        docs = sorted(book.get_items_of_type(ITEM_DOCUMENT), key=lambda x: x.file_name)

        for doc in docs:
            content = doc.get_content().decode("utf-8")
            soup = BeautifulSoup(content, "html.parser")

            # Try to find chapter divisions
            chapter_divs = soup.find_all(
                ["h1", "h2", "h3"], class_=lambda x: x and "chapter" in x.lower()
            )
            if not chapter_divs:
                chapter_divs = soup.find_all(
                    lambda tag: tag.name in ["h1", "h2", "h3"]
                    and (
                        "chapter" in tag.get_text().lower()
                        or "book" in tag.get_text().lower()
                    )
                )

            if chapter_divs:
                # Process each chapter division
                for i, div in enumerate(chapter_divs):
                    title = div.get_text().strip()

                    # Get content until next chapter heading or end
                    content = ""
                    for tag in div.find_next_siblings():
                        if tag.name in ["h1", "h2", "h3"] and (
                            "chapter" in tag.get_text().lower()
                            or "book" in tag.get_text().lower()
                        ):
                            break
                        content += tag.get_text() + "\n"

                    if content.strip():
                        chapters.append(
                            {
                                "title": title,
                                "content": content.strip(),
                                "order": len(chapters) + 1,
                            }
                        )
            else:
                # No chapter divisions found, treat whole document as one chapter
                text_content = soup.get_text().strip()
                if text_content:
                    # Try to find a title
                    title_tag = soup.find(["h1", "h2", "title"])
                    title = (
                        title_tag.get_text().strip()
                        if title_tag
                        else f"Chapter {len(chapters) + 1}"
                    )

                    if title.lower() not in [
                        "copy",
                        "copyright",
                        "title page",
                        "cover",
                    ]:
                        chapters.append(
                            {
                                "title": title,
                                "content": text_content,
                                "order": len(chapters) + 1,
                            }
                        )

    # Print summary
    if chapters:
        print("\nSuccessfully extracted {} chapters:".format(len(chapters)))
        for chapter in chapters:
            print(f"  {chapter['order']}. {chapter['title']}")

        total_words = sum(len(chapter["content"].split()) for chapter in chapters)
        print("\nBook Summary:")
        print(f"Total Chapters: {len(chapters)}")
        print(f"Total Words: {total_words:,}")
        print(f"Total Duration: {total_words / 150:.1f} minutes")
    else:
        print("\nWarning: No chapters were extracted!")

    return chapters
